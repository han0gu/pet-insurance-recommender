from langchain_core.documents import Document

chunk = Document(
    page_content=("최고의 심의기구를 말합니다.</td></tr></tbody></table><br><h1 id='137' "
 "style='font-size:16px'>\uf000 제1항의 수술에서 아래에 정한 사항은 제외합니다.</h1><br><p "
 "id='138' data-category='paragraph' style='font-size:16px'>1"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000448',
              'chunk_char_len': 181,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
