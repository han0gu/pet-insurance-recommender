from langchain_core.documents import Document

chunk = Document(
    page_content=('공정을 잃은 합의 사회통념상 일반 보통인이라면 그 같은 일을 하지 않을 정도로 현저하게 '
 "공정</td></tr></tbody></table><br><p id='173' data-category='paragraph' "
 "style='font-size:16px'>성을 잃은 것을 말합니다.</p><p id='174' "
 "data-category='paragraph' style='font-size:16px'>제49조(개인정보보호)</p><br><p "
 "id='175' data-category='list' style='font-size:16px'>\uf000"),
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
 'indexing': {'chunk_id': 'chunk_000322',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
