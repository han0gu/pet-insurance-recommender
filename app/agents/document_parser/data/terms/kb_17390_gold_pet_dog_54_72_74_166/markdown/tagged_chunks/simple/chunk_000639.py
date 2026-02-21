from langchain_core.documents import Document

chunk = Document(
    page_content=('- 22조(재가입)은 제외합니다.\n'
 '- \uf000 반려동물(강아지) 일반조항에서 정하지 않은 사항은 보통약관 제1절 일반조항을\n'
 '- 따릅니다. 다만, 이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금\n'
 '- 의 지급), 제24조(계약의 소멸) 및 제36조(중도인출)는 제외합니다.\n'
 '4. 반려동물장례비용지원금(실손)(30일면책)(강아지)【갱신계약】'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000639',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
