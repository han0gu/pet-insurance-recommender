from langchain_core.documents import Document

chunk = Document(
    page_content=('되는 항목</td><td>분류번호</td></tr><tr><td>우울에피소드</td><td>F32</td></tr><tr><td>재발성 '
 '우울장애</td><td>F33</td></tr><tr><td>공황장애[간헐 발작성 '
 '불안]</td><td>F41.0</td></tr></tbody></table> 외상후스트레스장애 주) 1'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001802',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
