from langchain_core.documents import Document

chunk = Document(
    page_content=(". 동<br>물</p><br><p id='201' data-category='paragraph' "
 "style='font-size:16px'>보험기간 중에 천식지속상태로 진단 확정된 경우</p><p id='202' "
 "data-category='paragraph' style='font-size:16px'>제2조(보험금 지급에 관한 "
 "세부규정)</p><br><p id='203' data-category='paragraph' "
 "style='font-size:16px'>\uf000 보험수익자와 회사가 제1조(보험금의</p><br><p id='204'"),
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
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
