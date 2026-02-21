from langchain_core.documents import Document

chunk = Document(
    page_content=('어 풀</td><td>이 현저하게 공정을 잃은 합의</td></tr><tr><td colspan="2">사회통념상 일반 보통인이라면 그 '
 '같은 일을 하지 않을 정도로 현저하게 공정 성을 잃은 것을 말합니다.</td></tr></tbody></table><br><h1 '
 "id='97' style='font-size:14px'>제22조(재가입)</h1><br><h1 id='98' "
 "style='font-size:14px'>\uf000</h1><br><p id='99' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000916',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
