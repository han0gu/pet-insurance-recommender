from langchain_core.documents import Document

chunk = Document(
    page_content=('‘보험개발원이 공시하는 월평균 정기예금이율’을 연단위 복리로 계산한<br>금액을 더하며, 나누어 지급할 금액을 일시에 지급하는 경우에는 '
 "‘보험개발원이 공시하<br>는 월평균 정기예금이율’을 연단위 복리로 할인한 금액을 지급합니다.</p><br><h1 id='84' "
 "style='font-size:14px'>【보험개발원이 공시하는 월평균 정기예금이율】</h1><br><p id='85' "
 "data-category='paragraph' style='font-size:14px'>현재 시점의 정기예금이율은 보험개발원"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000072',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
