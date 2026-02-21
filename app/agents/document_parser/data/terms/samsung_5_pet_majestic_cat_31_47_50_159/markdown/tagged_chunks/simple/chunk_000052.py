from langchain_core.documents import Document

chunk = Document(
    page_content=('하지 아니하거나 부실의 고지를 한 때에는 보험자는 그 사실을 안 날부터 1월내에, 계약을 체결한\n'
 '날부터 3년내에 한하여 계약을 해지할 수 있다. 그러나 보험자가 계약당시에 그 사실을 알았거나\n'
 '중대한 과실로 인하여 알지 못한 때에는 그러하지 아니하다.[상법 제651조의2(서면에 의한 질문의 효력)]\n'
 '보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다.제 17조 (상해보험계약 후 알릴 의무)① 계약자 또는 피보험자는 보험기간 중에 '
 '피보험자에게 다음 각 호의 변경이 발생한 경'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
