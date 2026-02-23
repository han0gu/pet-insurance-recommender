from langchain_core.documents import Document

chunk = Document(
    page_content=('지급보험금 결정에는 영향을 미치지 않습니다.# 제11조(보험금 받는 방법의 변경)- ① 계약자(보험금 지급사유 발생 후에는 보험수익자)는 '
 '회사의 사업방법서에서 정한 바에\n'
 '- 따라 보험금의 전부 또는 일부에 대하여 나누어 지급받거나 일시에 지급받는 방법으로\n'
 '- 변경할 수 있습니다.\n'
 '- ② 회사는 제1항에 따라 일시에 지급할 금액을 나누어 지급하는 경우에는 나중에 지급할\n'
 '- 금액에 대하여 ‘보험개발원이 공시하는 월평균 정기예금이율’을 연단위 복리로 계산한'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
