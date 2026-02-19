from langchain_core.documents import Document

chunk = Document(
    page_content=('8. 수렵, 투견, 경주 등과 수색, 마약탐지, 경계등의 특수목적으로 업무수행 및 훈련 중의 손해 9. 애완동물의 선천적, 유전적 질병에 '
 '의한 손해(보험개시 이전부터 객관적으로 인지할 수 있는 증 상을 포함합니다. 다만 보험기간 중 최초로 발견된 경우에는 당해 보험기간에 '
 '한하여 보상합니 다.)\n'
 '10. 대한민국 이외 지역에서 발생한 사고 및 손해 11. 수의사 자격이 없는 자의 치료행위로 인한 비용 및 그로 인하여 가중된 손해'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['head',
                             'dental',
                             'skin',
                             'joint',
                             'urinary',
                             'eye',
                             'digestive',
                             'other']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
