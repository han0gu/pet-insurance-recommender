from langchain_core.documents import Document

chunk = Document(
    page_content=('병\n'
 '추상장해, 신체의 기형이나 기능장해가 발생하여 그 원상회복을 목적으로 사고일\n'
 '로부터 2년 이내에 성형외과 전문의로부터 성형수술을 받은 경우 아래에 정한 금\n'
 '액을 상해흉터복원수술비로 보험수익자에게 하나의 사고에 대하여 500만원한도\n'
 '반\n'
 '로 지급합니다.\n'
 '려\n'
 '(보험가입금액 7만원 고정)해구분안면부동\n'
 '상지․하지 물수술 1cm당 7만원\n'
 '상해흉터복원수술비 수술 1cm당 14만원\n'
 '(단, 3cm이상의 경우에 한함)'),
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
