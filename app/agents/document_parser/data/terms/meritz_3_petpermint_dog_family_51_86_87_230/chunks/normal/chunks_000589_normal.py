from langchain_core.documents import Document

chunk = Document(
    page_content=('① 피보험자가 피해자에게 지급할 책임을 지는 법률상의 손해배상금(손해배상금을 지급함으로써 대위 취득할 것이 있을 때에는 그 가액을 '
 '뺍니다) ② 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 조정에 관한 비용 ③ 보험증권상 보상한도액내의 금액에 대한 '
 '공탁보증보험 료. 그러나 회사는 그러한 보증을 제공할 책임은 부담 하지 않습니다.\n'
 '【상법 제657조(보험사고발생의 통지의무)】\n'
 '① 보험계약자 또는 피보험자나 보험수익자는 보험사고의 발생을 안 때에는 지체없이 보험자에게 그 통지를 발 송하여야 한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 176},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000589',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
