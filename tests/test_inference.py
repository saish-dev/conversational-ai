import pytest
from core.inference import predict_intent_for_test

def test_greet():
    assert predict_intent_for_test('hi', 'DemoAccount') == 'greet'
    assert predict_intent_for_test('hello', 'DemoAccount') == 'greet'
    assert predict_intent_for_test('hey there', 'DemoAccount') == 'greet'

def test_bye():
    assert predict_intent_for_test('bye', 'DemoAccount') == 'bye'
    assert predict_intent_for_test('see you later', 'DemoAccount') == 'bye'

def test_ask_help():
    assert predict_intent_for_test('can you help me?', 'DemoAccount') == 'ask_help'
    assert predict_intent_for_test('i need assistance', 'DemoAccount') == 'ask_help'

def test_reset_password():
    assert predict_intent_for_test('reset my password', 'DemoAccount') == 'reset_password'
    assert predict_intent_for_test('forgot password', 'DemoAccount') == 'reset_password'

def test_ask_time():
    assert predict_intent_for_test('what time is it?', 'DemoAccount') == 'ask_time'
    assert predict_intent_for_test('time please', 'DemoAccount') == 'ask_time'

def test_ask_joke():
    assert predict_intent_for_test('tell me a joke', 'DemoAccount') == 'ask_joke'
    assert predict_intent_for_test('make me laugh', 'DemoAccount') == 'ask_joke'

def test_ask_info():
    assert predict_intent_for_test('what is your name?', 'DemoAccount') == 'ask_info'
    assert predict_intent_for_test('tell me about your services', 'DemoAccount') == 'ask_info' 