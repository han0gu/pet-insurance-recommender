from langchain_core.documents import Document

chunk = Document(
    page_content=('료의 납입최고(독촉)기간까지의 이자(보험계약대출이율 이내에서 회사가 별도로 정하\n'
 '는 이율을 적용하여 계산)를 더한 금액이 해당 보험료가 납입된 것으로 계산한 해약환\n'
 '급금과 계약자에게 지급할 기타 모든 지급금의 합계액에서 계약자의 회사에 대한 모\n'
 '든 채무액을 뺀 금액을 초과하는 경우에는 보험료의 자동대출납입을 더는 할 수 없습\n'
 '니다.<용어풀이># [보험계약대출이율]해당 보험상품의 약관에 따라 계약자가 대출을 받을 경우, 회사가 정하는 대출이율이며, 이 특별약'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
