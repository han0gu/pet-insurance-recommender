from langchain_core.documents import Document

chunk = Document(
    page_content=('<유의사항>\n'
 '계약자가 회사에 보험수익자가 변경되었음을 통지하기 전에 보험금 지급사유가 발생한 경우 회사는 변경 전 보험수익자에게 보험금을 지급할 수 '
 '있습니다. 회사가 변경 전 보험수익자에게 보험금을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 150,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
