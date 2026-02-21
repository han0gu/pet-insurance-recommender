from langchain_core.documents import Document

chunk = Document(
    page_content=("받은 경우 이 특별약관의 보험가입금액을 골절부목치료비로 보험수익</p><br><p id='30' "
 "data-category='paragraph' style='font-size:16px'>자에게 매 사고시마다 "
 "지급합니다.</p><br><h1 id='31' style='font-size:16px'>제2조(보험금 지급에 관한</h1><br><h1 "
 "id='32' style='font-size:16px'>세부규정)</h1><br><p id='33' data-category='list' "
 "style='font-size:16px'>\uf000"),
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
 'indexing': {'chunk_id': 'chunk_000558',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
