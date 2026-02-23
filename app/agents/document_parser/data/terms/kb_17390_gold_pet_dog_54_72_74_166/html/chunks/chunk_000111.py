from langchain_core.documents import Document

chunk = Document(
    page_content=('중</td><td>직업의 변경으로 위험이 증가(상해급수 2급)되었으나,</td><td></td><td>1급 → '
 '이를</td></tr><tr><td colspan="4">회사에 알리지 않고 변경전 보험료를 계속 납입하던 중 상해사망 사고가 발생한 '
 '경우 ∙상해사망 가입금액 : 1억원 ∙상해사망 보험요율 : 1급 0.3, 2급 0.5 → 고객이 수령하는 상해사망 보험금 = 1억원 × '
 '(0.3 ÷ 0.5) = 6천만원 \uf000 계약자 또는 피보험자가 고의 또는 중대한 과실로 제1항 각 호의 변경사실을'),
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
