from langchain_core.documents import Document

chunk = Document(
    page_content=('인이 되는 위반사실을 구체적으로 명시)뿐만 아니라 계약 전 알릴 의무사항이 중요한 사항에 해당되는 사유를 "반대증거가 있는 경우 이의를 '
 '제기할 수 있습니다"라는 문구 와 함께 계약자에게 서면 또는 전자문서 등으로 알려 드립니다. 회사가 전자문서로 안 내하고자 할 경우에는 '
 '계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 전 자서명으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다. '
 '계약자 의 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 72},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000397',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
