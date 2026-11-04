"""Harvia Sauna Sensor (harvia.io device type SaunaSensor) measurements."""
import json
import math

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.harvia_sauna.api_harviaio import (
    _normalize_state_payload,
    _normalize_telemetry_payload,
)
from custom_components.harvia_sauna.const import API_PROVIDER_HARVIAIO, DOMAIN
from custom_components.harvia_sauna.coordinator import (
    HarviaDeviceData,
    _apply_state_data,
    _apply_telemetry_data,
    battery_low,
    presence_detected,
)
from custom_components.harvia_sauna.websocket_harviaio import HarviaIoWebSocketManager

from .test_setup import DEVICE_ID, _setup

# One minute frame from a real Sauna Sensor (firmware 3.0.130), identifiers removed
SAUNA_SENSOR_FRAME = {
    "tempEsp": 66, "temp": 80.548, "hum": 8.601, "resetCnt": 3, "targetTemp": 88,
    "saunaStatus": 2, "totalHours": 2, "batteryVoltage": 10.605,
    "batteryLoadV": 10.136, "batteryMedianV": 10.136, "presence": 749,
    "heapSize": 84980, "rssi": -59, "targetReached": True, "timeToTarget": 0,
}


def _entity_id(hass: HomeAssistant, platform: str, key: str) -> str:
    entity_id = er.async_get(hass).async_get_entity_id(
        platform, DOMAIN, f"{DEVICE_ID}_{key}"
    )
    assert entity_id is not None, f"no {platform} entity for {key}"
    return entity_id


def test_frame_fields_are_normalized() -> None:
    normalized = _normalize_telemetry_payload({"data": SAUNA_SENSOR_FRAME})
    assert normalized["timeToTarget"] == 0
    assert normalized["targetReached"] is True
    assert normalized["presence"] == 749
    assert normalized["batteryVoltage"] == 10.605
    assert normalized["wifiRSSI"] == -59


def test_state_thresholds_are_normalized() -> None:
    normalized = _normalize_state_payload(
        "d", {"PIRThreshold": 20, "batteryWarningThreshold": 8.5, "saunaStatus": 0}
    )
    assert normalized["PIRThreshold"] == 20
    assert normalized["batteryWarningThreshold"] == 8.5


async def test_sauna_sensor_frame_reaches_the_entities(hass: HomeAssistant) -> None:
    entry, api = await _setup(hass, API_PROVIDER_HARVIAIO)
    coordinator = hass.data[DOMAIN][entry.entry_id]
    manager = HarviaIoWebSocketManager(api, coordinator._async_handle_ws_update)

    # Before the sensor reports anything the new entities are unknown
    for platform, key in (
        ("sensor", "time_to_target"),
        ("sensor", "battery_voltage"),
        ("binary_sensor", "target_reached"),
        ("binary_sensor", "occupancy"),
        ("binary_sensor", "battery_low"),
    ):
        assert hass.states.get(_entity_id(hass, platform, key)).state == "unknown"

    await manager._handle_message("data", {"payload": {"data": {
        "devicesMeasurementsUpdateFeed": {"item": {
            "deviceId": DEVICE_ID, "data": json.dumps(SAUNA_SENSOR_FRAME),
        }}}}})
    await hass.async_block_till_done()

    device = coordinator.data.devices[DEVICE_ID]
    assert device.reported_time_to_target == 0
    assert device.presence == 749
    assert device.wifi_rssi == -59

    assert float(hass.states.get(_entity_id(hass, "sensor", "time_to_target")).state) == 0
    assert float(
        hass.states.get(_entity_id(hass, "sensor", "battery_voltage")).state
    ) == pytest.approx(10.605)
    assert hass.states.get(_entity_id(hass, "binary_sensor", "target_reached")).state == "on"
    assert hass.states.get(_entity_id(hass, "binary_sensor", "occupancy")).state == "on"
    assert hass.states.get(_entity_id(hass, "binary_sensor", "battery_low")).state == "off"

    # The raw presence level is a diagnostic sensor, disabled by default
    presence = er.async_get(hass).async_get(_entity_id(hass, "sensor", "presence"))
    assert presence.disabled_by is er.RegistryEntryDisabler.INTEGRATION


def test_time_to_target_is_kept_whatever_the_run_state() -> None:
    """A heater switched by an external relay never reports itself active."""
    device = HarviaDeviceData(device_id="d")
    assert not device.active
    _apply_telemetry_data(device, {"timeToTarget": 12})
    assert device.reported_time_to_target == 12


@pytest.mark.parametrize("bad", ["n/a", None, math.nan, math.inf, True, {"v": 1}])
def test_unusable_values_read_as_unknown(bad) -> None:
    device = HarviaDeviceData(device_id="d")
    _apply_telemetry_data(
        device, {"timeToTarget": 5, "presence": 30, "batteryVoltage": 10.0}
    )
    _apply_telemetry_data(
        device, {"timeToTarget": bad, "presence": bad, "batteryVoltage": bad}
    )
    assert device.reported_time_to_target is None
    assert device.presence is None
    assert device.battery_voltage is None


def test_negative_time_to_target_reads_as_unknown() -> None:
    device = HarviaDeviceData(device_id="d")
    _apply_telemetry_data(device, {"timeToTarget": -1})
    assert device.reported_time_to_target is None


def test_target_reached_accepts_numbers_and_none() -> None:
    device = HarviaDeviceData(device_id="d")
    _apply_telemetry_data(device, {"targetReached": 1})
    assert device.target_reached is True
    _apply_telemetry_data(device, {"targetReached": 0})
    assert device.target_reached is False
    _apply_telemetry_data(device, {"targetReached": None})
    assert device.target_reached is None


def test_occupancy_and_battery_use_the_defaults_until_the_state_reports() -> None:
    device = HarviaDeviceData(device_id="d")
    assert presence_detected(device) is None
    assert battery_low(device) is None

    _apply_telemetry_data(device, {"presence": 20, "batteryVoltage": 8.5})
    assert presence_detected(device) is True  # default PIRThreshold 20
    assert battery_low(device) is True  # default warning 8.5 V

    _apply_telemetry_data(device, {"presence": 19, "batteryVoltage": 8.6})
    assert presence_detected(device) is False
    assert battery_low(device) is False


def test_occupancy_and_battery_follow_the_reported_thresholds() -> None:
    device = HarviaDeviceData(device_id="d")
    _apply_telemetry_data(device, {"presence": 749, "batteryVoltage": 10.605})
    assert presence_detected(device) is True
    assert battery_low(device) is False

    _apply_state_data(device, {"PIRThreshold": 800, "batteryWarningThreshold": 11})
    assert device.pir_threshold == 800
    assert device.battery_warning_threshold == 11
    assert presence_detected(device) is False
    assert battery_low(device) is True
