from langchain_core.documents import Document

chunk = Document(
    page_content=('골절</td><td>S42</td></tr><tr><td>아래팔의 골절 손목 및 손부위의 골절</td><td>S52 '
 'S62</td></tr><tr><td>대퇴골의 골절</td><td>S72</td></tr><tr><td>발목을 포함한 아래다리의 '
 '골절</td><td>S82</td></tr><tr><td>발목을 제외한 발의 '
 '골절</td><td>S92</td></tr><tr><td>여러 신체부위를 침범한 '
 '골절</td><td>T02</td></tr><tr><td>척추의 상세불명 부위의'),
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
 'indexing': {'chunk_id': 'chunk_001720',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
