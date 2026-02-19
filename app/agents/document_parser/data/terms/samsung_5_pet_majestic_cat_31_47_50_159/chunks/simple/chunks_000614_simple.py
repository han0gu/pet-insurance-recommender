from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 회사가 제1항에 의한 납입최고(독촉) 등을 전자문서로 안내하고자 할 경우에는 계약자 에게 서면, 전자서명법 제2조 제2호에 따른 '
 '전자서명으로 동의를 얻어 수신확인을 조 건으로 전자문서를 송신하여야 하며, 계약자가 전자문서에 대하여 수신을 확인하기 전까지는그 '
 '전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문서가 수신되지 않은 것을 확인한 경우에는 제1항에서 정한 내용을 서면(등기우편 등) '
 '또는 전화(음성 녹음)로 다시 알려 드립니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000614',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
