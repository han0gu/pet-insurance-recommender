from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(지정대리청구인의 지정)\n'
 '① 보험계약자는 보통약관 또는 특별약관에서 정한 보험금을 직접 청구할 수 없는 특별한 사정이 있을 경우를 대비하여 계약을 체결할 때 또는 '
 '계약체결 이후 다음 각 호의 1에 해당하는 자 중에서 보험금의 대리청구인(이하「지정대리청구인」이라 합니다)을 2인 이내(2인 지정시 '
 '대표대리인을 지정)에서 지정(제4조에 따른 변경지정 포함)할 수 있습 니다. 다만, 지정대리청구인은 보험금 청구 시에도 다음 각 호의 '
 '1에 해당하여야 합니 다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 42},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000228',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
