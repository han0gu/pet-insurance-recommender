from langchain_core.documents import Document

chunk = Document(
    page_content=('의원 및 조산원으로 나누어집니다.# 제8조 (보험금의 지급절차)① 회사는 제7조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 '
 '드리고 휴대전\n'
 '화 문자메시지 또는 전자우편 등으로 송부하며, 그 서류를 접수한 날부터 3영업일 이\n'
 '내에 보험금을 지급합니다.\n'
 '② 회사가 보험금 지급사유를 조사ㆍ확인하기 위해 필요한 기간이 제1항의 지급기일을\n'
 '초과할 것이 명백히 예상되는 경우에는 그 구체적 사유와 지급예정일 및 보험금 가지\n'
 '급 제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여 피보험자 또는 보험수'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000026',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
