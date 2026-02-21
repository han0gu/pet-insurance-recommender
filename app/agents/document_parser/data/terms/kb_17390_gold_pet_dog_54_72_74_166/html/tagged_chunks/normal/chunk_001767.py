from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2">아데노바이러스폐렴 특정</td><td>J12.0</td></tr><tr><td rowspan="3">바이러스성 '
 '폐렴</td><td>파라인플루엔자바이러스폐렴</td><td>J12.2</td></tr><tr><td>사람메타뉴모바이러스폐렴</td><td>J12.3</td></tr><tr><td></td><td></td></tr></tbody></table><p '
 "id='88' data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 '펫보험(강아지)(무배당)(26.01)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001767',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
