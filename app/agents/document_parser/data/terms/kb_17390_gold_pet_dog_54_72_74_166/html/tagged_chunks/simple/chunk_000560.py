from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전<br>문의 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 "전액 부담<br>합니다.</p><br><p id='34' data-category='paragraph' "
 "style='font-size:16px'>제3조(부목(Splint Cast)치료의 정의)</p><br><h1 id='35' "
 'style=\'font-size:16px\'>\uf000 이 특별약관에서 "부목(Splint</h1><br><p id=\'36\' '
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000560',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
