from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자, 피보험자 또는 이들의 법정대리인의 고의로 생긴 손해에 대한 배상책임 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, '
 '노동쟁의 기타 이들과 유사한 사태로 생긴 손해에 대 한 배상책임 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변으로 생긴 '
 '손해에 대한 배상책임 4. 피보험자가 소유, 사용 또는 관리하는 재물이 손해를 입었을 경우에 그 재물에 대하여 정당한 권리를 가진 '
 '사람에게 부담하는 손해에 대한 배상책임 5'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000136',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
