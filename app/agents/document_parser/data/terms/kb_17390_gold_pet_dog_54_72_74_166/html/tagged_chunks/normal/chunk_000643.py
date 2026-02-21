from langchain_core.documents import Document

chunk = Document(
    page_content=('특별약관의 보험가입금액 50%</td><td>이 특별약관의 보험가입금액 100%</td></tr></tbody></table><p '
 "id='180' data-category='paragraph' style='font-size:16px'>제2조(보험금 지급에 "
 "관한<br>\uf000 보험수익자와 회사가</p><br><p id='181' data-category='paragraph' "
 "style='font-size:16px'>세부규정)</p><br><p id='182' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000643',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
