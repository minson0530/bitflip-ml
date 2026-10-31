import struct

# Format codes for the struct module:
#   '>'  big-endian byte order (most significant byte first)
#   'f'  a 32-bit float (FP32)
#   'I'  a 32-bit unsigned integer
FLOAT32 = '>f'
UINT32 = '>I'

SAMPLE_WEIGHTS = [0.05, -0.3, 1.5, 3.0]   # typical small weights, plus values >= 2


def float_to_bits(value):
    """Return the 32 raw bits of an FP32 number, as an integer."""
    raw_bytes = struct.pack(FLOAT32, value)    # the number as 4 bytes in memory
    (bits,) = struct.unpack(UINT32, raw_bytes)  # the same 4 bytes read as an integer
    return bits


def bits_to_float(bits):
    """Turn 32 raw bits back into an FP32 number."""
    raw_bytes = struct.pack(UINT32, bits)        # the integer as 4 bytes
    (value,) = struct.unpack(FLOAT32, raw_bytes)  # the same 4 bytes read as a float
    return value


def flip_bit(value, position):
    """Flip one bit of an FP32 number and return the new number.

    position 0 is the rightmost bit (least significant mantissa bit),
    position 31 is the sign bit.
    """
    bits = float_to_bits(value)
    mask = 1 << position          # a 1 at `position`, 0 everywhere else
    flipped_bits = bits ^ mask    # XOR flips only the bit where mask is 1
    return bits_to_float(flipped_bits)


def field_name(position):
    """Name the FP32 field that a bit position belongs to."""
    if position == 31:
        return "sign"
    if position >= 23:
        return "exponent"
    return "mantissa"


def format_bits(value):
    """Show the 32 bits of an FP32 number as 'sign exponent mantissa'."""
    bits = f"{float_to_bits(value):032b}"
    sign, exponent, mantissa = bits[0], bits[1:9], bits[9:]
    return f"{sign} {exponent} {mantissa}"


def print_flip_table(value):
    """Flip each of the 32 bits of `value` in turn and print the result."""
    print(f"\n=== original: {value}")
    print(f"    bits:     {format_bits(value)}")
    print(f"    {'bit':>3}  {'field':<8}  new value")
    for position in range(31, -1, -1):
        new_value = flip_bit(value, position)
        print(f"    {position:>3}  {field_name(position):<8}  {new_value}")


def main():
    for weight in SAMPLE_WEIGHTS:
        print_flip_table(weight)


if __name__ == "__main__":
    main()