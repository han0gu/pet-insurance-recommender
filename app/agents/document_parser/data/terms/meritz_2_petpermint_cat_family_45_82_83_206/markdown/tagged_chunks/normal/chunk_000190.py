from langchain_core.documents import Document

chunk = Document(
    page_content=('자가 회사에 권리를 대항하기 위해서는 계약자가 보험수익\n'
 '자가 변경되었음을 회사에 통지하여야 합니다.# 【부가설명】계약자가 보험수익자가 변경되었음을 회사에 통지하기\n'
 '전에 보험금 지급사유가 발생한 경우 회사는 변경 전 보\n'
 '험수익자에게 보험금을 지급할 수 있습니다. 회사가 변\n'
 '경 전 보험수익자에게 보험금을 지급한 경우 변경된 보\n'
 '험수익자에게는 별도로 보험금을 지급하지 않습니다.\uf000 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이96상 지난 유효한 '
 '계약으로서 그 보험종목의 변경을 요청할'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000190',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
