from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 같습니다)에서 정한 80%이상 장해지급률에 해당하는 장해상태가 되<br>었을 때에는 최초 1회에 한하여 이 보장의 보험가입금액 '
 "전액을 일반상해80%이상후</p><br><h1 id='191' style='font-size:14px'>유장해보험금으로 보험수익자에게 "
 "지급합니다.</h1><p id='192' data-category='list' style='font-size:14px'>제2조(보험금 "
 '지급에 관한 세부규정)<br>\uf000 제1조(보험금의 지급사유)에서 장해지급률이 상해 발생일부터 180일 이내에 확정<br>되지 않는'),
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
 'indexing': {'chunk_id': 'chunk_000336',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
