from langchain_core.documents import Document

chunk = Document(
    page_content=('발톱 질환</td></tr><tr><td>아토피성 피부염</td></tr><tr><td>알러지성 '
 '피부염</td></tr><tr><td>지간 '
 '피부염</td></tr><tr><td>지루</td></tr><tr><td>지방조직염</td></tr><tr><td>피하 농양 / 봉와직염 '
 '호산구성 육아종</td></tr><tr><td>기타 세균성 피부염</td></tr><tr><td>기타 '
 '피부염</td></tr><tr><td>기타 피부질환</td></tr><tr><td>기타 선천성 '
 '피부질환</td></tr><tr><td>원인 불명의 귀'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_001819',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
