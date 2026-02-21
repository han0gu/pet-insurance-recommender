from langchain_core.documents import Document

chunk = Document(
    page_content=('- 전까지는그 전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문서가 수신되지\n'
 '- 않은 것을 확인한 경우에는 제1항에서 정한 내용을 서면(등기우편 등) 또는 전화(음성\n'
 '- 녹음)로 다시 알려 드립니다.\n'
 '- ⑥ 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때 다음\n'
 '- 각 호의 요건을 모두 충족하는 경우에 「보험업감독규정」 제4-36조 제3항에 따른\n'
 '- 전자적 상품설명장치를 활용할 수 있습니다.\n'
 '1.계약자에게 전자적 상품설명장치를 활용하여 제1항에 따른 납입최고(독촉) 등을 한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000516',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
