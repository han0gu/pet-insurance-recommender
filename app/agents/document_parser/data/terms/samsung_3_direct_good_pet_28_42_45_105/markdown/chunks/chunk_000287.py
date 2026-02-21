from langchain_core.documents import Document

chunk = Document(
    page_content=('있는 모든 것을 포함합니다. 또한 음식물의 상태(부패, 감염 여부 등)와 상관없이 모두 포함됩니다.- 2. 질병: 상해를 제외한 상병을 '
 '모두 포함합니다.\n'
 '- 3. 보험가입금액 : 회사와 계약자간에 약정한 금액으로 보험사고가 발생할 때 회사가\n'
 '- 지급할 최대 보험금을 말합니다.\n'
 '- 4. 자기부담금 : 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부\n'
 '- 담하는 일정 금액을 말합니다.\n'
 '- 5. 보험금 분담 : 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약('),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
