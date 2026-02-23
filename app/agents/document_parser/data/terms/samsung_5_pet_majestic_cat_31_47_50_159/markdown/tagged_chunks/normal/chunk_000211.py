from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 경우 납입최고(독촉)와 특별약관의 해지)에서 정한 특별약관의 해지가 발생하지 않\n'
 '- 은 경우를 말합니다.\n'
 '- ⑨ 제30조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))에서 정한 특별\n'
 '- 약관의 부활이 이루어진 경우 부활을 청약한 날을 제6항의 청약일로 하여 적용합니\n'
 '- 다.\n'
 '# 제20조 (청약의 철회)① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만,\n'
 '회사가 건강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계약 또는 전문금융'),
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
 'indexing': {'chunk_id': 'chunk_000211',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
