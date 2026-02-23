from langchain_core.documents import Document

chunk = Document(
    page_content=('- 터)성형술, 레이저치료 등 포함)을 시행한 후에도\n'
 '- 영구히 남게 되는 상태의 추상(추한 모습)을 말한다.\n'
 '- 3) “추상(추한 모습)을 남긴 때”라 함은 상처의 흔적,\n'
 '- 화상 등으로 피부의 변색, 모발의 결손, 조직(뼈, 피\n'
 '- 부 등)의 결손 및 함몰 등으로 성형수술을 하여도 더\n'
 '- 이상 추상(추한 모습)이 없어지지 않는 경우를 말한\n'
 '- 다.\n'
 '- 4) 다발성 반흔(흉터) 발생시 각 판정부위(얼굴, 머리,\n'
 '- 목) 내의 다발성 반흔(흉터)의 길이 또는 면적은 합\n'
 '- 산하여 평가한다. 단, 길이가 5mm 미만의 반흔(흉'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000620',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
