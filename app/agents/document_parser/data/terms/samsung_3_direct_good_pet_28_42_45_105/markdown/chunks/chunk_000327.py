from langchain_core.documents import Document

chunk = Document(
    page_content=('급 제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여 피보험자 또는 보험수\n'
 '익자에게 즉시 통지합니다. 다만, 지급예정일은 다음 각 호의 어느 하나에 해당하는\n'
 '경우를 제외하고는 제8조(보험금의 청구)에서 정한 서류를 접수한 날부터 30영업일\n'
 '이내에서 정합니다.- \n'
 '1. 소송제기\n'
 '2. 분쟁조정 신청\n'
 '3. 수사기관의 조사\n'
 '4. 해외에서 발생한 보험사고에 대한 조사\n'
 '5. 제6항에 따른 회사의 조사요청에 대한 동의 거부 등 계약자, 피보험자 또는 보험수\n'
 '익자의 책임있는 사유로 보험금 지급사유의 조사 및 확인이 지연되는 경우'),
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
