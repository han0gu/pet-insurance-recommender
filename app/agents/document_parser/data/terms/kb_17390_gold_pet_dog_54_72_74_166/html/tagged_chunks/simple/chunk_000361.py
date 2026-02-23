from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중<br>에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 "전액 부담합니다.</p><br><p id='19' data-category='paragraph' "
 "style='font-size:14px'>74 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='20' "
 "data-category='list' style='font-size:14px'>제3조(특별약관의 소멸)<br>\uf000 회사는 "
 '제1조(보험금의 지급사유)에서 정한'),
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
 'indexing': {'chunk_id': 'chunk_000361',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
