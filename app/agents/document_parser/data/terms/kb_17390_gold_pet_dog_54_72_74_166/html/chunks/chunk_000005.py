from langchain_core.documents import Document

chunk = Document(
    page_content=('및 보상</td><td>관련 용어</td></tr></thead><tbody><tr><td>용 어 상해</td><td>정 의 보험기간 '
 '중에 발생한 급격하고도 우연한 외래의 사고로 신 체(의수, 의족, 의안, 의치 등 신체보조장구는 제외하나, 인공장기나 부분 의치 등 '
 '신체에 이식되어 그 기능을 대신 할 경우는 포함합니다)에 입은 상해를 '
 '말합니다.</td></tr><tr><td>장해</td><td>【별표1】(장해분류표)에서 정한 기준에 따른 장해상태를 '
 '말합니다.</td></tr><tr><td>중요한 사항</td><td>계약전 알릴'),
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
