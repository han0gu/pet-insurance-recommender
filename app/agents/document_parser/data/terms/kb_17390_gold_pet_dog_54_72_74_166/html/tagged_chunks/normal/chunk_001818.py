from langchain_core.documents import Document

chunk = Document(
    page_content=('외이도염</td></tr><tr><td>중이염</td></tr><tr><td>개선충 감염(옴감염)</td></tr><tr><td>벼룩 '
 '감염</td></tr><tr><td>진드기 감염 '
 '비만세포종</td></tr><tr><td>조직구종</td></tr><tr><td>흑색종</td></tr><tr><td>각화이상</td></tr><tr><td>농피증</td></tr><tr><td>다리 '
 '부위 피부염</td></tr><tr><td>두드러기</td></tr><tr><td>면역매개성피부질환</td></tr><tr><td>모낭염 '
 '발톱'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_001818',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
