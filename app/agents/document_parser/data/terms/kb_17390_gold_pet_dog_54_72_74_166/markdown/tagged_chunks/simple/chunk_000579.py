from langchain_core.documents import Document

chunk = Document(
    page_content=('| 특정처치 (이물제거) | 이물제거(내시경) | 이물섭취 |\n'
 '| 특정처치 (이물제거) | 이물제거(구토유도약물) | 이물섭취 |\n'
 '| 창상/교상치료 | 창상/교상치료 | 상해로 인한 창상 또는 교상 |\n'
 '| 특정약물치료Ⅱ | 특정약물치료Ⅱ | 상해 또는 질병 |\n'
 '| 특정재활치료Ⅱ | 특정재활치료Ⅱ | 상해 또는 질병 |\n'
 '| 항암약물치료 | 항암약물치료 | 암 |\n'
 '\uf000 반려동물이 제1항의 사고로 치료를 받던 중에 이 특별약관의 보험기간이 만료된- 경우에도 만료일부터 180일 이내의 의료비는 '
 '보상하여 드립니다. 다만, 사고일'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000579',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
