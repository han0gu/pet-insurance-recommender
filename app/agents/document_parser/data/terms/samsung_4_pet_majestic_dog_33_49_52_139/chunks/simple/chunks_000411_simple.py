from langchain_core.documents import Document

chunk = Document(
    page_content=('1 -7. 5대골절 수술비 특별약관\n'
 '제 1관 일반사항\n'
 '① 제2관 개별사항에서 정하지 않은 사항은 특별약관의 일반사항을 적용합니다. ② 제1항에도 불구하고, 이 상품의 사업방법서 별지에 따라 '
 '납입면제를 적용하지 않거나 보통약관에서 보험료 납입면제에 대해 정하지 않은 경우 특별약관 일반사항 제5조(보 험료 납입면제) 및 '
 '제6조(보험료 납입면제에 관한 세부규정)를 적용하지 않습니다. ③ 제1항에도 불구하고, 특별약관 일반사항 제35조(해약환급금)를 적용하지 '
 '않으며 보통 약관 제36조(해약환급금)을 적용합니다.\n'
 '제2관 개별사항'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000411',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
