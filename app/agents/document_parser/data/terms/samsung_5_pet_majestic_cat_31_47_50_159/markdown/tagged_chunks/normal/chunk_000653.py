from langchain_core.documents import Document

chunk = Document(
    page_content=('를 따릅니다. 이 경우 부활(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유)\n'
 '제3항을 적용합니다.부활(효력회복)되는 특별약관의 보장개시는 4-1. 반려묘 의료비(치과및 구강질 환포 함)(재# 제8조 (특별약관의 '
 '소멸)보험증권에 기재된 반려묘가 보험기간 중에 사망하여 이 추가특별약관에서 정한 보험금\n'
 '지급사유가 더이상 발생할 수 없는 경우에는 "보험료 및 해약환급금 산출방법서" 에 정\n'
 '하는 바에 따라 회사가 적립한 사망당시 이 추가특별약관의 계약자적립액 및 미경과보험'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000653',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
