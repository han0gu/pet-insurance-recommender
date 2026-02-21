from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list'></p><br><table id='78' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>용 어 풀</td><td>이 "
 '신의료기술평가위원회</td></tr><tr><td colspan="2">의료법 제54조(신의료기술평가위원회의 설치 등)에 의거 설치된 '
 "위원회로서 신의료기술에 관한 최고의 심의기구를 말합니다.</td></tr></tbody></table><br><h1 id='79' "
 "style='font-size:16px'>\uf000 제1항의 수술에서 아래에"),
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
 'indexing': {'chunk_id': 'chunk_000410',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
