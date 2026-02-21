from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소<br>속 전문의 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 "전액<br>부담합니다.</p><h1 id='234' style='font-size:14px'>제3조(특별약관의</h1><br><p "
 "id='235' data-category='paragraph' style='font-size:14px'>소멸)</p><br><p "
 "id='236' data-category='paragraph' style='font-size:14px'>피보험자가 사망하였을 경우에는 이"),
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
 'indexing': {'chunk_id': 'chunk_000505',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
