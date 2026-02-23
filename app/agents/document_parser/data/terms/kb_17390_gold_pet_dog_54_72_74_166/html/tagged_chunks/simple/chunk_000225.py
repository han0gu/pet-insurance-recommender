from langchain_core.documents import Document

chunk = Document(
    page_content=(". 또한, 보장개시일을 계약일로 봅니다.</td></tr></tbody></table><br><p id='34' "
 "data-category='paragraph' style='font-size:14px'>\uf000 회사는 제2항에도 불구하고 다음 중 "
 "한 가지에 해당되는 경우에는 보장을 하지 않<br>습니다.</p><br><p id='35' data-category='list' "
 "style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_000225',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
