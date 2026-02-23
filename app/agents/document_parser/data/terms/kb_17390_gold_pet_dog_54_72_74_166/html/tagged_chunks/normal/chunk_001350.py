from langchain_core.documents import Document

chunk = Document(
    page_content=('. 기타, 피보험자 또는 지정대리청구인이 보험금의 수령에 필요하여 제출하는 서류<br>\uf000 병원 또는 의원에서 제1항 제2호의 '
 '사고증명서를 발급받을 경우, 그 병원 또는 의<br>원은 의료법 제3조(의료기관)에서 정하는 국내의 병원이나 의원 또는 이와 '
 "동등하<br>질<br>다고 인정되는 국외의 의료기관이어야 합니다.</p><h1 id='227' "
 "style='font-size:16px'>제7조(보험금의 지급절차)</h1><br><p id='228' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001350',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
