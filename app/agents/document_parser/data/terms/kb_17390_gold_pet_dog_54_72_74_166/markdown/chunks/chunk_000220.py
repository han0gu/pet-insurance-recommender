from langchain_core.documents import Document

chunk = Document(
    page_content=('. 규정 | 예금자보호제도란 예금보험공사에서 금융기관 등으로부터 미리 보험료를 받아 적 립해 두었다가 금융기관이 경영악화나 파산 등으로 '
 '예금을 지급할 수 없는 경우 해당 금융기관을 대신하여 해약환급금(또는 만기시 보험금)에 기타지급금을 합한 법 금액 및 사고보험금을 각각 '
 '보험계약자 1인당 최고 1억원까지 지급함으로써 예금 ㆍ 자를 보호하는 제도를 말합니다. 규정 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
