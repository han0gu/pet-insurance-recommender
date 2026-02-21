from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>에 따라 법률상의 권리를 행사할 수 있습니다.</p><br><h1 id='80' "
 "style='font-size:14px'>용 어 풀</h1><br><p id='81' data-category='paragraph' "
 "style='font-size:14px'>이</p><br><p id='82' data-category='paragraph' "
 "style='font-size:14px'>∙</p><br><p id='83' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000269',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
