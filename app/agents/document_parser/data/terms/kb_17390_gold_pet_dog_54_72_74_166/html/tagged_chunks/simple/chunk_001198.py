from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제</td></tr></tbody></table><br><p id='217' data-category='paragraph' "
 "style='font-size:14px'>해</p><h1 id='218' "
 "style='font-size:16px'>제15조(대위권)</h1><br><h1 id='219' "
 "style='font-size:16px'>\uf000 회사가 보험금을</h1><br><p id='220' "
 "data-category='paragraph' style='font-size:16px'>지급한 때(현물보상한 경우를 포함합니다)에는 "
 '회사는'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001198',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
