from langchain_core.documents import Document

chunk = Document(
    page_content=("주입하는 것</td></tr></tbody></table><br><p id='173' data-category='paragraph' "
 "style='font-size:14px'>상</p><br><h1 id='174' "
 "style='font-size:16px'>제5조(특별약관의 소멸)</h1><br><p id='175' "
 "data-category='paragraph' style='font-size:14px'>해</p><p id='176' "
 "data-category='list' style='font-size:14px'>\uf000 보험증권에 기재된"),
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
 'indexing': {'chunk_id': 'chunk_000981',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
