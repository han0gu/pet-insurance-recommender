from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 피보험자의 가족관계등록상의 배우자 2. 피보험자의 3촌 이내의 친족\n'
 '② 제1항에도 불구하고 지정대리청구인이 지정된 이후에 제1조(적용대상)의 보험수익자 가 변경되는 경우에는 이미 지정된 지정대리청구인의 '
 '자격은 자동적으로 상실된 것으 로 봅니다. ③ 제1항에도 불구하고 보험계약에서 지정대리청구인의 지정 기간을 별도로 제한한 경 우, '
 '계약자는 이 특별약관에서도 그 기간에 한하여 지정대리청구인을 지정할 수 있습 니다.\n'
 '제4조 (지정대리청구인의 변경지정)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000634',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
