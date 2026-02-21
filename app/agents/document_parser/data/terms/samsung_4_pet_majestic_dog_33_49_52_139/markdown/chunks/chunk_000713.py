from langchain_core.documents import Document

chunk = Document(
    page_content=('납입하여야 합니다. 이러한 경우 피보험자에게 보험사고가 발생하였을 때에는 보\n'
 '험계약에 정한 보험금을 지급합니다.# <용어풀이># [할증위험률에 의한 보험료]피보험자의 건강상태가 회사가 정한 기준에 적합하지 않은 '
 '경우 일반위험률보다 높은 위험률을 적\n'
 '용하여 산출된 보험료를 말합니다.[표준체 보험료]\n'
 '할증위험률의 가입조건(보험기간, 납입기간, 피보험자의 가입나이 등)과 동일한 기준에서, 일반위# 험률을 적용하여 산출된 보험료를 '
 '말합니다.# 2. 보험금감액법계약일부터 회사가 정하는 삭감기간 내에 보험계약의 규정에 정하는 상해 이외의'),
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
