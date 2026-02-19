from langchain_core.documents import Document

chunk = Document(
    page_content=('- 70 -\n'
 '제1관 일반사항\n'
 '① 제2관 개별사항에서 정하지 않은 사항은 특별약관의 일반사항을 적용합니다. ② 제1항에도 불구하고, 이 상품의 사업방법서 별지에 따라 '
 '납입면제를 적용하지 않거나 보통약관에서 보험료 납입면제에 대해 정하지 않은 경우 특별약관 일반사항 제5조(보 험료 납입면제) 및 '
 '제6조(보험료 납입면제에 관한 세부규정)를 적용하지 않습니다. ③ 제1항에도 불구하고, 특별약관 일반사항 제35조(해약환급금)를 적용하지 '
 '않으며 보통 약관 제36조(해약환급금)을 적용합니다.\n'
 '제2관 개별사항\n'
 '제1조 (보험금의 지급사유)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 71},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000364',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
