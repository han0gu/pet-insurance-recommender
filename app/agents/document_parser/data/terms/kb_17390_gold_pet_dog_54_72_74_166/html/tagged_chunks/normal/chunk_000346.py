from langchain_core.documents import Document

chunk = Document(
    page_content=('후 질병으로 오른쪽 눈의 교정시력이 0.1이하(지급률15%)인 상태 에서 이후 상해로 그 오른쪽 눈의 교정시력이 0.02이하가 된 '
 '경우(지급률 35%) ⇒ 장해지급률 35%에서 질병으로 인한 장해지급률 15%를 차감한 지급률 20%(=35%-15%)에 해당하는 '
 "후유장해보험금을 지급</td></tr></tbody></table><h1 id='196' "
 "style='font-size:14px'>제3조(보장의 소멸)</h1><br><p id='197' data-category='list' "
 "style='font-size:14px'>\uf000"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000346',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
