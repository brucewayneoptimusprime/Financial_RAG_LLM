from app.macro_utils import latest_value, latest_yoy

print("CPI latest level:", latest_value("cpi"))
print("CPI latest YoY:", latest_yoy("cpi"))
print("Unemployment latest:", latest_value("unemployment"))
